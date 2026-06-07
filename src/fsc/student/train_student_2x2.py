"""
Full 2x2 KD ablation with TUNED hyperparameters, multi-seed.

Factorial design (capacity x data), each cell x 4 methods x 3 seeds:

              100% data        20% data
  strong   strong_100        strong_20
  small    small_100         small_20

Methods (tuned on the small student, see outputs/student/kd_hparam_sweep.csv):
  T = 8, lambda_logit = 1.0, lambda_feature = 1.0
  ce_only    : CE
  logit_kd   : CE + 1.0 * KL(T=8)
  feature_kd : CE + 1.0 * cosine
  full_kd    : CE + 1.0 * KL(T=8) + 1.0 * cosine

Seeds = [42, 43, 44]. For the 20% settings, the seed drives BOTH the stratified
data subset AND the init (so each seed = a different 20% subset + different init).
For 100% settings the seed only drives init. Report mean +/- std over seeds.

Selection per run = best-by-val-macro-F1 (val = dev set; FSC test untouched).

Writes (NEW location, existing results untouched):
  outputs/student/kd_2x2_tuned/results.csv     (all 48 runs, long form)
  outputs/student/kd_2x2_tuned/summary.csv     (16 cells, mean +/- std)
  outputs/student/kd_2x2_tuned/kd_2x2_tuned.png
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
    load_data, evaluate, spec_augment, kd_logit_loss, kd_feature_loss,
    stratified_indices, EPOCHS, LR, WEIGHT_DECAY, BATCH_SIZE, LABEL_SMOOTH, DEVICE,
)

OUT = OUTPUTS_ROOT / "fsc" / "student" / "kd_2x2_tuned"
OUT.mkdir(parents=True, exist_ok=True)

# ── Tuned KD hyperparameters ──────────────────────────────────────────────────────
T_KD        = 8.0
LAM_LOGIT   = 1.0
LAM_FEATURE = 1.0
SEEDS       = [42, 43, 44]

# method -> (lam_logit, lam_feature)
METHODS = {
    "ce_only":    (0.0, 0.0),
    "logit_kd":   (LAM_LOGIT, 0.0),
    "feature_kd": (0.0, LAM_FEATURE),
    "full_kd":    (LAM_LOGIT, LAM_FEATURE),
}

SMALL_KW = {"channels": (16, 32, 64, 96, 128), "proj_hidden": None}


def strong_factory():
    return DSResNetSE()


def small_factory():
    return DSResNetSE(**SMALL_KW)


# setting -> (model_factory, data_fraction)
SETTINGS = [
    ("strong_100", strong_factory, 1.00),
    ("strong_20",  strong_factory, 0.20),
    ("small_100",  small_factory,  1.00),
    ("small_20",   small_factory,  0.20),
]


def run(model_factory, train_data, val_data, lam_logit, lam_feature, seed, frac):
    torch.manual_seed(seed)
    np.random.seed(seed)

    Xtr, ytr, ztr, ltr = train_data
    if frac < 1.0:
        sub = torch.as_tensor(stratified_indices(ytr, frac, seed), dtype=torch.long)
        Xtr, ytr, ztr, ltr = Xtr[sub], ytr[sub], ztr[sub], ltr[sub]
    Xva, yva, _, _ = val_data

    model = model_factory().to(DEVICE)
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
                loss = loss + lam_logit * kd_logit_loss(logits_s, ltr[idx].to(DEVICE), T_KD)
            if lam_feature > 0:
                loss = loss + lam_feature * kd_feature_loss(z_s, ztr[idx].to(DEVICE).float())
            loss.backward()
            opt.step()
        sched.step()
        m = evaluate(model, Xva, yva)
        if m["macro_f1"] > best["macro_f1"]:
            best = m
    return best


def main():
    print(f"Device: {DEVICE}")
    print(f"strong params {model_summary(strong_factory())['params']:,}  "
          f"| small params {model_summary(small_factory())['params']:,}")
    print(f"KD: T={T_KD:g}, lam_logit={LAM_LOGIT:g}, lam_feature={LAM_FEATURE:g}  | seeds={SEEDS}\n")

    train_data, val_data = load_data()

    runs = []   # long form: one row per (setting, method, seed)
    for setting, factory, frac in SETTINGS:
        print(f"\n##### {setting} (frac={frac}) #####")
        for method, (ll, lf) in METHODS.items():
            for seed in SEEDS:
                m = run(factory, train_data, val_data, ll, lf, seed, frac)
                runs.append({"setting": setting, "method": method, "seed": seed,
                             "T": T_KD, "lam_logit": ll, "lam_feature": lf,
                             "val_acc": round(m["acc"], 4),
                             "macro_f1": round(m["macro_f1"], 4),
                             "weighted_f1": round(m["weighted_f1"], 4)})
                print(f"  {setting:<10} {method:<11} seed{seed}: "
                      f"acc={m['acc']:.4f} macroF1={m['macro_f1']:.4f}")

    # ── results.csv (all runs) ────────────────────────────────────────────────────
    res_path = OUT / "results.csv"
    with open(res_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["setting", "method", "seed", "T", "lam_logit",
                                          "lam_feature", "val_acc", "macro_f1", "weighted_f1"])
        w.writeheader(); w.writerows(runs)
    print(f"\nResults CSV -> {res_path}")

    # ── summary.csv (mean +/- std over seeds) ─────────────────────────────────────
    summary = []
    for setting, _, _ in SETTINGS:
        for method in METHODS:
            cell = [r for r in runs if r["setting"] == setting and r["method"] == method]
            row = {"setting": setting, "method": method, "n_seeds": len(cell)}
            for k in ("val_acc", "macro_f1", "weighted_f1"):
                vals = np.array([r[k] for r in cell])
                row[f"{k}_mean"] = round(float(vals.mean()), 4)
                row[f"{k}_std"] = round(float(vals.std()), 4)
            summary.append(row)
    sum_path = OUT / "summary.csv"
    with open(sum_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["setting", "method", "n_seeds",
                                          "val_acc_mean", "val_acc_std",
                                          "macro_f1_mean", "macro_f1_std",
                                          "weighted_f1_mean", "weighted_f1_std"])
        w.writeheader(); w.writerows(summary)
    print(f"Summary CSV -> {sum_path}")

    # console summary
    print(f"\n{'setting':<11}{'method':<12}{'macroF1 mean±std':<20}{'gain vs CE'}")
    print("-" * 60)
    for setting, _, _ in SETTINGS:
        ce_mean = next(r["macro_f1_mean"] for r in summary if r["setting"] == setting and r["method"] == "ce_only")
        for method in METHODS:
            r = next(x for x in summary if x["setting"] == setting and x["method"] == method)
            gain = r["macro_f1_mean"] - ce_mean
            print(f"{setting:<11}{method:<12}{r['macro_f1_mean']:.4f}±{r['macro_f1_std']:.4f}     {gain:+.4f}")

    make_plot(summary)
    print("\nDone.")


def make_plot(summary):
    method_color = {"ce_only": "#7f7f7f", "logit_kd": "#1f77b4",
                    "feature_kd": "#ff7f0e", "full_kd": "#2ca02c"}
    methods = list(METHODS.keys())
    # 2x2 grid: rows = capacity (strong top / small bottom), cols = data (100% left / 20% right)
    grid = [["strong_100", "strong_20"], ["small_100", "small_20"]]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for r in range(2):
        for c in range(2):
            ax = axes[r][c]
            setting = grid[r][c]
            means = [next(x["macro_f1_mean"] for x in summary if x["setting"] == setting and x["method"] == m) for m in methods]
            stds = [next(x["macro_f1_std"] for x in summary if x["setting"] == setting and x["method"] == m) for m in methods]
            x = np.arange(len(methods))
            ax.bar(x, means, yerr=stds, capsize=4, color=[method_color[m] for m in methods])
            ce = means[0]
            ax.axhline(ce, color="gray", linestyle=":", linewidth=1.2, label=f"CE-only ({ce:.3f})")
            for xi, mv in zip(x, means):
                ax.text(xi, mv, f"{mv:.3f}", ha="center", va="bottom", fontsize=8)
            lo = min(means) - max(stds) - 0.015
            hi = max(means) + max(stds) + 0.015
            ax.set_ylim(lo, hi)
            ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=8, rotation=12)
            ax.set_title(setting, fontsize=11)
            ax.legend(fontsize=8, loc="lower right"); ax.grid(axis="y", alpha=0.3)
            if c == 0:
                ax.set_ylabel("Macro F1 (FSC val)")
    fig.suptitle("2x2 KD Ablation (tuned: T=8, lam_logit=1.0, lam_feature=1.0; 3 seeds, mean±std)\n"
                 "rows = capacity (strong/small)  ·  cols = data (100%/20%)  ·  dotted = CE-only",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "kd_2x2_tuned.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot        -> {OUT / 'kd_2x2_tuned.png'}")


if __name__ == "__main__":
    main()
