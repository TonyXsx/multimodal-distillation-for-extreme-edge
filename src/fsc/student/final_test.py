"""
Final held-out TEST evaluation (README_test.md).

All validation-based selection is frozen. We train the SMALL DSResNet-SE student
once per method on the TRAINING set, select the best checkpoint on VAL, then
evaluate that checkpoint ONCE on the held-out FSC TEST split. The student is
audio-only, so TEST needs no teacher signal.

Fixed configuration (from the val ablations / 2x2 tuned run):
  student = small DSResNet-SE (98K); T = 8; lam_logit = 1.0; lam_feature = 1.0
  feature loss = cosine; train recipe (epochs/lr/wd/batch/specaug/label-smooth)
  inherited unchanged from kd_common.

Methods: CE-only, Logit-KD, Feature-KD, Full-KD. Single run each (this is the
final model report; statistical significance comes from the multi-seed val
ablations, not from this single TEST pass).

Prereqs: data/student/logmel_cache/test_logmel.pt (precompute_logmel.py).

Outputs:
  data/student/final_test_checkpoints/<method>.pt
  outputs/student/final_test/results.csv      (source of truth)
  outputs/student/final_test/final_test.png
"""

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from common.config import OUTPUTS_ROOT                              # noqa: E402
from common.models.audio_student import DSResNetSE, model_summary  # noqa: E402
from fsc.student.kd_common import (                                # noqa: E402
    DATA, LOGMEL, load_data, evaluate, spec_augment, kd_logit_loss, kd_feature_loss,
    EPOCHS, LR, WEIGHT_DECAY, BATCH_SIZE, LABEL_SMOOTH, SEED, DEVICE,
)

CKPT_DIR = DATA / "student" / "final_test_checkpoints"
OUT      = OUTPUTS_ROOT / "fsc" / "student" / "final_test"
for d in (CKPT_DIR, OUT):
    d.mkdir(parents=True, exist_ok=True)

# Fixed KD config (same as the 2x2 tuned run)
T_KD        = 8.0
LAM_LOGIT   = 1.0
LAM_FEATURE = 1.0

SMALL_KW = {"channels": (16, 32, 64, 96, 128), "proj_hidden": None}

# method -> (lam_logit, lam_feature)
METHODS = {
    "ce_only":    (0.0, 0.0),
    "logit_kd":   (LAM_LOGIT, 0.0),
    "feature_kd": (0.0, LAM_FEATURE),
    "full_kd":    (LAM_LOGIT, LAM_FEATURE),
}


def load_test():
    """Test log-mel normalized with TRAIN mean/std (same as val); audio-only."""
    tr = torch.load(LOGMEL / "train_logmel.pt", weights_only=False)
    te = torch.load(LOGMEL / "test_logmel.pt", weights_only=False)
    mean = tr["mean"].view(1, 1, 1, -1)
    std  = tr["std"].view(1, 1, 1, -1)
    Xte = (te["logmel"].float() - mean) / std
    yte = te["labels"].long()
    return Xte, yte


def run(method, train_data, val_data, Xte, yte):
    ll, lf = METHODS[method]
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    Xtr, ytr, ztr, ltr = train_data
    Xva, yva, _, _ = val_data

    model = DSResNetSE(**SMALL_KW).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    ce = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

    n = Xtr.shape[0]
    best_f1, best_state, best_val, best_epoch = -1.0, None, None, -1
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb = spec_augment(Xtr[idx].to(DEVICE))
            yb = ytr[idx].to(DEVICE)
            opt.zero_grad()
            z_s, logits_s = model(xb)
            loss = ce(logits_s, yb)
            if ll > 0:
                loss = loss + ll * kd_logit_loss(logits_s, ltr[idx].to(DEVICE), T_KD)
            if lf > 0:
                loss = loss + lf * kd_feature_loss(z_s, ztr[idx].to(DEVICE).float())
            loss.backward()
            opt.step()
        sched.step()

        m_val = evaluate(model, Xva, yva)               # VAL: checkpoint selection only
        if m_val["macro_f1"] > best_f1:
            best_f1 = m_val["macro_f1"]
            best_val = m_val
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Evaluate the selected checkpoint ONCE on the held-out TEST split.
    model.load_state_dict(best_state)
    m_test = evaluate(model, Xte, yte)

    torch.save({"method": method, "state_dict": best_state, "best_epoch": best_epoch,
                "val_metrics": best_val, "test_metrics": m_test,
                "T": T_KD, "lam_logit": ll, "lam_feature": lf},
               CKPT_DIR / f"{method}.pt")
    return best_val, m_test, best_epoch


def main():
    print(f"Device: {DEVICE}")
    sz = model_summary(DSResNetSE(**SMALL_KW))
    print(f"Small student params: {sz['params']:,}  | FP32 {sz['fp32_mb']:.2f} MB  | INT8(est) {sz['int8_mb']:.2f} MB")
    print(f"KD: T={T_KD:g}, lam_logit={LAM_LOGIT:g}, lam_feature={LAM_FEATURE:g}  | seed={SEED}\n")

    train_data, val_data = load_data()
    Xte, yte = load_test()
    print(f"test {tuple(Xte.shape)}\n")

    results = []
    for method in METHODS:
        print(f"=== {method} ===")
        val_m, test_m, best_epoch = run(method, train_data, val_data, Xte, yte)
        print(f"  selected @epoch {best_epoch+1} (val macroF1={val_m['macro_f1']:.4f})  ->  "
              f"TEST acc={test_m['acc']:.4f} macroF1={test_m['macro_f1']:.4f} wF1={test_m['weighted_f1']:.4f}")
        results.append({"method": method,
                        "test_acc": round(test_m["acc"], 4),
                        "test_macro_f1": round(test_m["macro_f1"], 4),
                        "test_weighted_f1": round(test_m["weighted_f1"], 4),
                        "val_macro_f1_sel": round(val_m["macro_f1"], 4),
                        "best_epoch": best_epoch + 1,
                        "params": sz["params"], "fp32_mb": round(sz["fp32_mb"], 2),
                        "int8_mb": round(sz["int8_mb"], 2)})

    # ── results CSV (source of truth) ─────────────────────────────────────────────
    csv_path = OUT / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["method", "test_acc", "test_macro_f1", "test_weighted_f1",
                                          "val_macro_f1_sel", "best_epoch", "params", "fp32_mb", "int8_mb"])
        w.writeheader(); w.writerows(results)
    print(f"\nResults CSV -> {csv_path}")

    ce = results[0]
    print(f"\n{'method':<12}{'TEST acc':<10}{'TEST macroF1':<14}{'gain vs CE (macroF1)'}")
    print("-" * 56)
    for r in results:
        print(f"{r['method']:<12}{r['test_acc']:<10.4f}{r['test_macro_f1']:<14.4f}"
              f"{r['test_macro_f1'] - ce['test_macro_f1']:+.4f}")

    make_plot(results)
    print("Done.")


def make_plot(results):
    methods = [r["method"] for r in results]
    metrics = [("test_acc", "Test Accuracy", "#1f77b4"),
               ("test_macro_f1", "Test Macro F1", "#ff7f0e"),
               ("test_weighted_f1", "Test Weighted F1", "#2ca02c")]
    vals_all = [r[k] for r in results for k, _, _ in metrics]
    lo, hi = min(vals_all) - 0.012, max(vals_all) + 0.012

    fig, ax = plt.subplots(figsize=(11, 6.5))
    x = np.arange(len(methods))
    w = 0.26
    for j, (key, label, color) in enumerate(metrics):
        vals = [r[key] for r in results]
        bars = ax.bar(x + (j - 1) * w, vals, w, label=label, color=color)
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{b.get_height():.4f}",
                    ha="center", va="bottom", fontsize=8, rotation=90)
    ce_mf1 = results[0]["test_macro_f1"]
    ax.axhline(ce_mf1, color="gray", linestyle=":", linewidth=1.3,
               label=f"CE-only macroF1 ({ce_mf1:.4f})")
    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylim(lo, hi)
    ax.set_ylabel("Score (FSC TEST holdout, 31 classes)")
    ax.set_title("FINAL TEST — small DSResNet-SE (98K) on held-out FSC test\n"
                 "T=8, lam_logit=1.0, lam_feature=1.0  ·  single final run per method  ·  (y-axis zoomed)",
                 fontsize=10)
    ax.legend(fontsize=9, loc="lower right"); ax.grid(axis="y", alpha=0.35)
    fig.savefig(OUT / "final_test.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot        -> {OUT / 'final_test.png'}")


if __name__ == "__main__":
    main()
