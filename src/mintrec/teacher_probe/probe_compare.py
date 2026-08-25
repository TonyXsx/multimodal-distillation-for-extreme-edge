"""
Compares the frozen-teacher feature sets with a sweep over head capacity, and
plots it.

Finds every data/mintrec/teacher_features/mintrec2.0*/ folder that has train and
dev features. Each feature key gets several heads trained on train and evaluated
on dev, test untouched. Standardised on train stats, fixed 50 epochs,
AdamW(1e-3, wd 1e-4), batch 256, CE, nothing selects on dev.

Heads in capacity order: linear, 1024, 2048-1024, plus dropout 0.5 versions of
the MLPs to see whether depth or regularisation is the lever.
"""

import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from common.config import MINTREC_DATA, MINTREC_OUTPUTS  # noqa: E402
from common.probe import Probe                           # noqa: E402

EPOCHS, LR, WEIGHT_DECAY, BATCH_SIZE, SEED = 50, 1e-3, 1e-4, 256, 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HEADLINE = "pf_audio_mean_L24-27-30-34"

# (id, hidden_dims, dropout, label), in capacity order for the x-axis
ARCHS = [
    ("A1", [],           0.1, "linear"),
    ("A2", [1024],       0.1, "1024"),
    ("A3", [2048, 1024], 0.1, "2048-1024"),
    ("A4", [1024],       0.5, "1024 (d.5)"),
    ("A5", [2048, 1024], 0.5, "2048-1024 (d.5)"),
]

OUT_DIR = MINTREC_OUTPUTS / "teacher_probe"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FEAT_ROOT = MINTREC_DATA / "teacher_features"


def standardize(Xtr, Xev):
    m = Xtr.mean(0, keepdim=True)
    s = Xtr.std(0, keepdim=True).clamp_min(1e-6)
    return (Xtr - m) / s, (Xev - m) / s


def train_one(hidden, dropout, Xtr, ytr, Xev, yev, n_classes):
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = Probe(Xtr.shape[1], hidden, n_classes, dropout=dropout).to(DEVICE)
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
        tr_acc = (model(Xtr_d).argmax(1) == ytr_d).float().mean().item()
        ev_pred = model(Xev_d).argmax(1)
    dev_acc = (ev_pred == yev_d).float().mean().item()
    dev_f1 = f1_score(yev.numpy(), ev_pred.cpu().numpy(), average="macro")
    return tr_acc, dev_acc, dev_f1


def variant_label(tag):
    if "pf_text-audio" in tag:
        return "ta: audio+text (no video)"
    if "bf16" in tag:
        return "bf16 tva (full video)"
    if "aware" in tag:
        return "4bit tva-aware"
    return "4bit tva-plain"


def main():
    dirs = sorted(d for d in FEAT_ROOT.glob("mintrec2.0*")
                  if (d / "train_features.pt").exists() and (d / "dev_features.pt").exists())
    assert dirs, f"no feature folders under {FEAT_ROOT}"
    print(f"Device: {DEVICE} | {len(dirs)} feature sets | {len(ARCHS)} heads\n")

    results = []
    for d in dirs:
        tag = d.name
        tr = torch.load(d / "train_features.pt", weights_only=False)
        dv = torch.load(d / "dev_features.pt", weights_only=False)
        ytr, yev = tr["labels"].long(), dv["labels"].long()
        n_classes = int(max(ytr.max(), yev.max()).item()) + 1
        print(f"=== {variant_label(tag)} ===")
        for feat in tr["features"].keys():
            Xtr_n, Xev_n = standardize(tr["features"][feat].float(), dv["features"][feat].float())
            for arch_id, hidden, drop, lab in ARCHS:
                tr_acc, dev_acc, dev_f1 = train_one(hidden, drop, Xtr_n, ytr, Xev_n, yev, n_classes)
                results.append({"variant": tag, "feature": feat, "arch": arch_id, "arch_desc": lab,
                                "dropout": drop, "dev_acc": dev_acc, "dev_macro_f1": dev_f1, "train_acc": tr_acc})
                if feat == HEADLINE:
                    print(f"  [{arch_id} {lab:<16}] dev_acc={dev_acc:.4f} macroF1={dev_f1:.4f} (train={tr_acc:.3f})")
        print()

    csv_path = OUT_DIR / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "feature", "arch", "arch_desc", "dropout",
                                          "dev_acc", "dev_macro_f1", "train_acc"])
        w.writeheader(); w.writerows(results)

    # dev acc / macro-F1 against head size, one line per feature set
    xids = [a[0] for a in ARCHS]
    xlabs = [f"{a[0]}\n{a[3]}" for a in ARCHS]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for metric, ax, title in zip(["dev_acc", "dev_macro_f1"], axes, ["dev accuracy", "dev macro-F1"]):
        for d in dirs:
            tag = d.name
            ys = [next(r[metric] for r in results if r["variant"] == tag
                       and r["feature"] == HEADLINE and r["arch"] == aid) for aid in xids]
            ax.plot(xids, ys, marker="o", label=variant_label(tag))
        ax.set_xticks(range(len(xids))); ax.set_xticklabels(xlabs, fontsize=8)
        ax.set_title(title); ax.set_ylabel(metric); ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("MIntRec2.0 teacher-probe: feature set × head capacity (headline feat, dev)", fontsize=11)
    fig.tight_layout()
    png = OUT_DIR / "probe_comparison.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"Results CSV -> {csv_path}\nPlot      -> {png}")


if __name__ == "__main__":
    main()
