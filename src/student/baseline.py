"""
Stronger CE-only baseline (no distillation) for the audio student.

Diagnosis recap (speaker-independent FSC split, 10 unseen val speakers):
the original student fit train to 100% but only reached 72% val — a 27.6pp
speaker-generalization gap caused by from-scratch training with NO augmentation
and weak regularization. This script tests what closes that gap, keeping the
parameter count ~same (~0.38M) — NOT a bigger model.

Two changes, decomposed so we can see which one matters:

  arch_only : architecture fixes only
              - less aggressive downsampling (freq kept at 8 bins, not 2)
              - BatchNorm on the 64-dim bottleneck + head (GPT's points 2/3)
              - dropout 0.2
  arch_reg  : arch fixes + regularization
              - SpecAugment (freq/time masking)  <- main lever for speaker generalization
              - label smoothing 0.1

Outputs:
  data/student/baseline/<name>_best.pt
  outputs/student/baseline_comparison.png
  outputs/student/baseline_results.md
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

PROJECT = Path(r"D:\msc_AI\individual_project\multimodal-distillation-for-extreme-edge")
sys.path.insert(0, str(PROJECT / "src" / "student"))
from student_model import ResDSSEBlock, model_summary   # noqa: E402  (reuse building blocks)

DATA   = PROJECT / "data"
LOGMEL = DATA / "student" / "logmel_cache"
CKPT   = DATA / "student" / "baseline"
OUT    = PROJECT / "outputs" / "student"
for d in (CKPT, OUT):
    d.mkdir(parents=True, exist_ok=True)

EPOCHS      = 70
LR          = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE  = 256
SEED        = 42
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"


# ── Revised architecture (params ~same as DSResNetSE) ────────────────────────────
class BaselineNet(nn.Module):
    """Same channel schedule as DSResNetSE; gentler frequency downsampling and a
    normalized projection head. Downsampling (T, F):
        301x64 -stem(2,2)-> 151x32 -b1(2,2)-> 76x16 -b2(2,2)-> 38x8
              -b3(2,1)-> 19x8 -b4(2,1)-> 10x8  (freq kept at 8, not 2)
    """

    def __init__(self, n_classes=31, proj_dim=64, dropout=0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=(2, 2), padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        self.block1 = ResDSSEBlock(32, 64, stride=(2, 2))
        self.block2 = ResDSSEBlock(64, 128, stride=(2, 2))
        self.block3 = ResDSSEBlock(128, 192, stride=(2, 1))
        self.block4 = ResDSSEBlock(192, 256, stride=(2, 1))

        self.head = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, proj_dim),
        )
        self.bottleneck_norm = nn.BatchNorm1d(proj_dim)   # normalize 64-dim bottleneck
        self.classifier = nn.Linear(proj_dim, n_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x); x = self.block2(x); x = self.block3(x); x = self.block4(x)
        x = x.mean(dim=(2, 3))
        z = self.bottleneck_norm(self.head(x))
        return z, self.classifier(z)


# ── SpecAugment (per-batch, training only) ───────────────────────────────────────
def spec_augment(x, n_freq=2, n_time=2, f_max=12, t_max=40):
    B, _, T, F = x.shape
    x = x.clone()
    for _ in range(n_freq):
        f = int(torch.randint(0, f_max + 1, (1,)))
        if f > 0:
            f0 = int(torch.randint(0, max(1, F - f), (1,)))
            x[:, :, :, f0:f0 + f] = 0.0
    for _ in range(n_time):
        t = int(torch.randint(0, t_max + 1, (1,)))
        if t > 0:
            t0 = int(torch.randint(0, max(1, T - t), (1,)))
            x[:, :, t0:t0 + t, :] = 0.0
    return x


# ── Data ──────────────────────────────────────────────────────────────────────
def load_data():
    tr = torch.load(LOGMEL / "train_logmel.pt", weights_only=False)
    va = torch.load(LOGMEL / "val_logmel.pt", weights_only=False)
    mean, std = tr["mean"].view(1, 1, 1, -1), tr["std"].view(1, 1, 1, -1)
    Xtr = (tr["logmel"].float() - mean) / std
    Xva = (va["logmel"].float() - mean) / std
    return Xtr, tr["labels"].long(), Xva, va["labels"].long()


@torch.no_grad()
def predict(model, X):
    model.eval()
    out = []
    for i in range(0, X.shape[0], BATCH_SIZE):
        _, lg = model(X[i:i + BATCH_SIZE].to(DEVICE))
        out.append(lg.argmax(1).cpu())
    return torch.cat(out).numpy()


# ── Train one config ─────────────────────────────────────────────────────────────
def train(cfg, Xtr, ytr, Xva, yva):
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = BaselineNet().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    ce = nn.CrossEntropyLoss(label_smoothing=cfg["label_smooth"])
    n = Xtr.shape[0]

    best = {"val_f1": -1}
    hist = {"train_acc": [], "val_acc": []}
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb = Xtr[idx].to(DEVICE)
            if cfg["specaug"]:
                xb = spec_augment(xb)
            yb = ytr[idx].to(DEVICE)
            opt.zero_grad()
            _, logits = model(xb)
            loss = ce(logits, yb)
            loss.backward(); opt.step()
        sched.step()

        ptr = predict(model, Xtr); pva = predict(model, Xva)
        a_tr = accuracy_score(ytr.numpy(), ptr)
        a_va = accuracy_score(yva.numpy(), pva)
        f_va = f1_score(yva.numpy(), pva, average="macro")
        hist["train_acc"].append(a_tr); hist["val_acc"].append(a_va)
        if f_va > best["val_f1"]:
            best = {"val_f1": f_va, "val_acc": a_va, "train_acc": a_tr, "epoch": epoch,
                    "weighted_f1": f1_score(yva.numpy(), pva, average="weighted"),
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:
            print(f"    epoch {epoch+1:3d}/{EPOCHS}  train_acc={a_tr:.4f}  val_acc={a_va:.4f}  macroF1={f_va:.4f}")

    torch.save({"cfg": cfg, "state_dict": best["state"], "best_epoch": best["epoch"],
                "val_acc": best["val_acc"], "val_macro_f1": best["val_f1"],
                "weighted_f1": best["weighted_f1"], "train_acc_at_best": best["train_acc"],
                "history": hist}, CKPT / f"{cfg['name']}_best.pt")
    return best, hist


CONFIGS = [
    {"name": "arch_only", "specaug": False, "label_smooth": 0.0},
    {"name": "arch_reg",  "specaug": True,  "label_smooth": 0.1},
]


def main():
    print(f"Device: {DEVICE}")
    sz = model_summary(BaselineNet())
    print(f"BaselineNet params: {sz['params']:,}  | FP32 {sz['fp32_mb']:.2f} MB  (target ~0.38M)\n")

    Xtr, ytr, Xva, yva = load_data()
    print(f"train {tuple(Xtr.shape)}  val {tuple(Xva.shape)}\n")

    results, hists = [], {}
    for cfg in CONFIGS:
        print(f"=== {cfg['name']}  (specaug={cfg['specaug']}, label_smooth={cfg['label_smooth']}) ===")
        best, hist = train(cfg, Xtr, ytr, Xva, yva)
        gap = best["train_acc"] - best["val_acc"]
        print(f"  BEST @epoch {best['epoch']+1}: val_acc={best['val_acc']:.4f}  "
              f"macroF1={best['val_f1']:.4f}  weightedF1={best['weighted_f1']:.4f}  "
              f"(train_acc={best['train_acc']:.4f}, gap={gap:.4f})\n")
        results.append({"name": cfg["name"], "val_acc": best["val_acc"], "macro_f1": best["val_f1"],
                        "weighted_f1": best["weighted_f1"], "train_acc": best["train_acc"], "gap": gap})
        hists[cfg["name"]] = hist

    # ── results MD ───────────────────────────────────────────────────────────────
    md = ["| Config | Val Acc | Macro F1 | Weighted F1 | Train Acc | Gap |",
          "| ------ | ------- | -------- | ----------- | --------- | --- |"]
    for r in results:
        md.append(f"| {r['name']} | {r['val_acc']:.4f} | {r['macro_f1']:.4f} | {r['weighted_f1']:.4f} "
                  f"| {r['train_acc']:.4f} | {r['gap']:.4f} |")
    md.append(f"\n(reference) original DSResNetSE CE-only: val_acc 0.7235, train_acc 1.0000, gap 0.2765, params 379,119")
    (OUT / "baseline_results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))

    # ── plot: train/val curves + final comparison ─────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.4, 1]})
    colors = {"arch_only": "#1f77b4", "arch_reg": "#2ca02c"}
    for name, h in hists.items():
        ep = range(1, len(h["val_acc"]) + 1)
        ax1.plot(ep, h["val_acc"], "-", color=colors[name], label=f"{name} val")
        ax1.plot(ep, h["train_acc"], "--", color=colors[name], alpha=0.5, label=f"{name} train")
    ax1.axhline(0.7235, color="gray", linestyle=":", label="orig CE-only val (0.7235)")
    ax1.axhline(0.80, color="red", linestyle=":", alpha=0.5, label="target 0.80")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("accuracy"); ax1.set_ylim(0, 1.02)
    ax1.set_title("Baseline training curves (solid=val, dashed=train)\n"
                  "smaller train-val gap = better generalization", fontsize=10)
    ax1.legend(fontsize=8, loc="lower right"); ax1.grid(alpha=0.3)

    names = [r["name"] for r in results]
    x = np.arange(len(names)); w = 0.35
    ax2.bar(x - w/2, [r["train_acc"] for r in results], w, label="train acc", color="#cccccc")
    ax2.bar(x + w/2, [r["val_acc"] for r in results], w, label="val acc", color="#2ca02c")
    for i, r in enumerate(results):
        ax2.text(i + w/2, r["val_acc"] + 0.005, f"{r['val_acc']:.3f}", ha="center", fontsize=9)
        ax2.text(i - w/2, r["train_acc"] + 0.005, f"{r['train_acc']:.3f}", ha="center", fontsize=8, color="gray")
    ax2.axhline(0.7235, color="gray", linestyle=":", label="orig val (0.7235)")
    ax2.set_xticks(x); ax2.set_xticklabels(names); ax2.set_ylim(0, 1.05)
    ax2.set_title("Train vs Val accuracy (gap)", fontsize=10)
    ax2.legend(fontsize=8, loc="lower right"); ax2.grid(axis="y", alpha=0.3)

    fig.savefig(OUT / "baseline_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot -> {OUT / 'baseline_comparison.png'}")
    print("Done.")


if __name__ == "__main__":
    main()
