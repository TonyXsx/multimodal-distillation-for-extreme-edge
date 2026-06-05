"""
Cheap KD hyperparameter sweep (greedy coordinate descent) on the SMALL student
+ 100% data, single seed. KD has real signal on the constrained small student,
so an optimum is actually detectable here (unlike strong+100% where KD ~= 0).

Selection is on val (dev set); FSC test stays untouched for the final report.

Search:
  1. CE-only                                              (anchor)
  2. Logit-KD: T in {2,4,8} at lam_logit=0.5  -> pick best T  (+ boundary probe T=16,
              recorded right after T8 but excluded from selection)
              then best T at lam_logit in {0.1, 1.0}      -> pick best lam_logit (vs 0.5)
              (+ boundary probe lam_logit=3.0, recorded after ll0.1, excluded from selection)
  3. Feature-KD: lam_feature in {0.3, 1.0, 3.0}           -> pick best lam_feature
  4. Full-KD: (best T, best lam_logit, best lam_feature)
              optional: (best T, lam_logit=0.1, best lam_feature)

Boundary probes sit inline in their sweep group (so the CSV/plot read in sweep
order) but are kept out of the argmax so selection stays over the original grid.

Results written by code to (source of truth = results.csv):
  outputs/student/hparam_sweep/results.csv
  outputs/student/hparam_sweep/sweep.png
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

PROJECT = Path(r"D:\msc_AI\individual_project\multimodal-distillation-for-extreme-edge")
sys.path.insert(0, str(PROJECT / "src" / "student"))
from student_model import DSResNetSE, model_summary           # noqa: E402
from kd_common import (                                       # noqa: E402
    load_data, evaluate, spec_augment, kd_logit_loss, kd_feature_loss,
    EPOCHS, LR, WEIGHT_DECAY, BATCH_SIZE, LABEL_SMOOTH, SEED, DEVICE,
)

OUT = PROJECT / "outputs" / "student" / "hparam_sweep"
OUT.mkdir(parents=True, exist_ok=True)

SMALL_KW = {"channels": (16, 32, 64, 96, 128), "proj_hidden": None}


def small_factory():
    return DSResNetSE(**SMALL_KW)


# ── One training run with explicit KD hyperparameters ────────────────────────────
def run(train_data, val_data, t, lam_logit, lam_feature, seed=SEED):
    Xtr, ytr, ztr, ltr = train_data
    Xva, yva, _, _ = val_data
    torch.manual_seed(seed); np.random.seed(seed)

    model = small_factory().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    ce = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

    n = Xtr.shape[0]
    best = {"macro_f1": -1.0}
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
            if lam_logit > 0:
                loss = loss + lam_logit * kd_logit_loss(logits_s, ltr[idx].to(DEVICE), t)
            if lam_feature > 0:
                loss = loss + lam_feature * kd_feature_loss(z_s, ztr[idx].to(DEVICE).float())
            loss.backward(); opt.step()
        sched.step()
        m = evaluate(model, Xva, yva)
        if m["macro_f1"] > best["macro_f1"]:
            best = m
    return best


# ── Sweep ────────────────────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")
    print(f"Small student params: {model_summary(small_factory())['params']:,}")
    train_data, val_data = load_data()

    rows = []

    def record(stage, name, t, ll, lf, m):
        rows.append({"stage": stage, "name": name,
                     "T": t, "lam_logit": ll, "lam_feature": lf,
                     "val_acc": round(m["acc"], 4),
                     "macro_f1": round(m["macro_f1"], 4),
                     "weighted_f1": round(m["weighted_f1"], 4)})
        print(f"  [{stage}] {name}: acc={m['acc']:.4f} macroF1={m['macro_f1']:.4f} wF1={m['weighted_f1']:.4f}")
        return m

    # 1. CE-only anchor
    print("\n# 1. CE-only")
    m_ce = record("ce_only", "ce_only", "", 0.0, 0.0,
                  run(train_data, val_data, t=2.0, lam_logit=0.0, lam_feature=0.0))

    # 2a. Logit-KD: T sweep at lam_logit=0.5
    print("\n# 2a. Logit-KD T sweep (lam_logit=0.5)")
    logit_T = {}
    for T in (2.0, 4.0, 8.0):
        logit_T[T] = record("logit_T_sweep", f"T{T:g}_ll0.5", T, 0.5, 0.0,
                            run(train_data, val_data, t=T, lam_logit=0.5, lam_feature=0.0))
    best_T = max(logit_T, key=lambda k: logit_T[k]["macro_f1"])
    print(f"  -> best T = {best_T:g} (selection over {{2,4,8}})")
    # boundary probe one step past the T grid: recorded right after T8, NOT in selection
    record("logit_T_sweep", "T16_ll0.5", 16.0, 0.5, 0.0,
           run(train_data, val_data, t=16.0, lam_logit=0.5, lam_feature=0.0))

    # 2b. Logit-KD: lam_logit sweep at best T (0.5 already known from 2a)
    print(f"\n# 2b. Logit-KD lam_logit sweep (T={best_T:g})")
    logit_ll = {0.5: logit_T[best_T]}
    logit_ll[0.1] = record("logit_ll_sweep", f"T{best_T:g}_ll0.1", best_T, 0.1, 0.0,
                           run(train_data, val_data, t=best_T, lam_logit=0.1, lam_feature=0.0))
    # boundary probe one step past the lam_logit grid: after ll0.1, NOT in selection
    record("logit_ll_sweep", f"T{best_T:g}_ll3", best_T, 3.0, 0.0,
           run(train_data, val_data, t=best_T, lam_logit=3.0, lam_feature=0.0))
    logit_ll[1.0] = record("logit_ll_sweep", f"T{best_T:g}_ll1", best_T, 1.0, 0.0,
                           run(train_data, val_data, t=best_T, lam_logit=1.0, lam_feature=0.0))
    best_ll = max(logit_ll, key=lambda k: logit_ll[k]["macro_f1"])
    print(f"  -> best lam_logit = {best_ll:g} (selection over {{0.1,0.5,1.0}})")

    # 3. Feature-KD: lam_feature sweep
    print("\n# 3. Feature-KD lam_feature sweep")
    feat = {}
    for lf in (0.3, 1.0, 3.0):
        feat[lf] = record("feature_sweep", f"lf{lf:g}", "", 0.0, lf,
                         run(train_data, val_data, t=2.0, lam_logit=0.0, lam_feature=lf))
    best_lf = max(feat, key=lambda k: feat[k]["macro_f1"])
    print(f"  -> best lam_feature = {best_lf:g}")

    # 4. Full-KD with best of each, plus optional smaller lam_logit
    print("\n# 4. Full-KD")
    record("full_kd", f"full_T{best_T:g}_ll{best_ll:g}_lf{best_lf:g}", best_T, best_ll, best_lf,
           run(train_data, val_data, t=best_T, lam_logit=best_ll, lam_feature=best_lf))
    if best_ll != 0.1:
        record("full_kd_optional", f"full_T{best_T:g}_ll0.1_lf{best_lf:g}", best_T, 0.1, best_lf,
               run(train_data, val_data, t=best_T, lam_logit=0.1, lam_feature=best_lf))

    # ── write results CSV (source of truth) ───────────────────────────────────────
    csv_path = OUT / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["stage", "name", "T", "lam_logit", "lam_feature",
                                          "val_acc", "macro_f1", "weighted_f1"])
        w.writeheader(); w.writerows(rows)
    print(f"\nResults CSV -> {csv_path}")

    print("\n=== Selected hyperparameters (by val macro-F1, from the in-grid sweep) ===")
    print(f"  best T          = {best_T:g}")
    print(f"  best lam_logit  = {best_ll:g}")
    print(f"  best lam_feature= {best_lf:g}")
    print(f"  CE-only macroF1 = {m_ce['macro_f1']:.4f}")

    make_plot(rows, m_ce["macro_f1"])
    print("Done.")


def make_plot(rows, ce_f1):
    names = [r["name"] for r in rows]
    f1 = [r["macro_f1"] for r in rows]
    stage_color = {"ce_only": "#7f7f7f", "logit_T_sweep": "#1f77b4", "logit_ll_sweep": "#17becf",
                   "feature_sweep": "#ff7f0e", "full_kd": "#2ca02c", "full_kd_optional": "#98df8a"}
    colors = [stage_color.get(r["stage"], "#aaaaaa") for r in rows]

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(names))
    ax.bar(x, f1, color=colors)
    ax.axhline(ce_f1, color="gray", linestyle=":", linewidth=1.3, label=f"CE-only ({ce_f1:.3f})")
    for xi, v in zip(x, f1):
        ax.text(xi, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=7.5, rotation=40, ha="right")
    lo = min(f1 + [ce_f1]) - 0.01; hi = max(f1) + 0.015
    ax.set_ylim(lo, hi)
    ax.set_ylabel("Val Macro F1 (FSC, 31 classes)")
    ax.set_title("KD hyperparameter sweep — SMALL student + 100% data (single seed, val selection)\n"
                 "greedy: T@λ_logit=0.5 → λ_logit@bestT → λ_feature; then Full-KD; + boundary probes", fontsize=10)
    ax.legend(fontsize=9, loc="lower right"); ax.grid(axis="y", alpha=0.3)
    fig.savefig(OUT / "sweep.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot        -> {OUT / 'sweep.png'}")


if __name__ == "__main__":
    main()
