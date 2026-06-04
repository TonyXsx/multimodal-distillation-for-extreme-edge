"""
Focused KD ablations: does KD help more when the student is data- or
capacity-limited? (See README_limited_ablation.md.)

Two new settings, same KD hyperparameters / losses / training recipe as the
main full-data strong-student run (reused from train_student.py):

  Setting 1: strong student + 20% training data   (3 seeds, mean +/- std)
  Setting 2: small  student + 100% training data   (single seed 42)

The main full-data strong-student result (Table 1) is loaded from the existing
data/student/results.csv produced by train_student.py — not re-run.

Only train + val are used. The test set stays untouched.

Outputs:
  data/student/limited_ablation_results.csv
  outputs/student/limited_ablation.png
  outputs/student/limited_ablation_results.md
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
from student_model import DSResNetSE, model_summary          # noqa: E402
from train_student import (                                  # noqa: E402  (reuse, no main() runs on import)
    load_data, evaluate, spec_augment, kd_logit_loss, kd_feature_loss,
    EXPERIMENTS, USE_LOGIT, USE_FEATURE,
    EPOCHS, LR, WEIGHT_DECAY, BATCH_SIZE, DROPOUT, LABEL_SMOOTH,
    LAM_LOGIT, LAM_FEATURE, DEVICE,
)

DATA = PROJECT / "data"
OUT  = PROJECT / "outputs" / "student"
OUT.mkdir(parents=True, exist_ok=True)

DATA_FRAC = 0.20
SEEDS_20PCT = [42, 43, 44]
SMALL_KW = {"channels": (16, 32, 64, 96, 128), "proj_hidden": None}


# ── Stratified subset ─────────────────────────────────────────────────────────────
def stratified_indices(labels, frac, seed):
    rng = np.random.RandomState(seed)
    y = labels.numpy()
    idx = []
    for c in np.unique(y):
        c_idx = np.where(y == c)[0]
        k = max(1, int(round(len(c_idx) * frac)))
        idx.extend(rng.choice(c_idx, size=k, replace=False))
    return np.sort(np.array(idx))


# ── Train one (model_factory, optional subset, seed, exp) ────────────────────────────
def run(model_factory, train_data, val_data, exp, seed, subset_idx=None):
    Xtr, ytr, ztr, ltr = train_data
    Xva, yva, _, _ = val_data
    if subset_idx is not None:
        sub = torch.as_tensor(subset_idx, dtype=torch.long)
        Xtr, ytr, ztr, ltr = Xtr[sub], ytr[sub], ztr[sub], ltr[sub]

    use_logit, use_feature = USE_LOGIT[exp], USE_FEATURE[exp]
    torch.manual_seed(seed); np.random.seed(seed)

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
            if use_logit:
                loss = loss + LAM_LOGIT * kd_logit_loss(logits_s, ltr[idx].to(DEVICE))
            if use_feature:
                loss = loss + LAM_FEATURE * kd_feature_loss(z_s, ztr[idx].to(DEVICE).float())
            loss.backward(); opt.step()
        sched.step()
        m = evaluate(model, Xva, yva)
        if m["macro_f1"] > best["macro_f1"]:
            best = m
    return best


# ── Settings ─────────────────────────────────────────────────────────────────────
def setting_strong_20pct(train_data, val_data):
    print("\n##### Setting 1: STRONG student + 20% data (3 seeds) #####")
    per_exp = {e: {"acc": [], "macro_f1": [], "weighted_f1": []} for e in EXPERIMENTS}
    for seed in SEEDS_20PCT:
        sub = stratified_indices(train_data[1], DATA_FRAC, seed)
        print(f"  seed {seed}: subset n={len(sub)}")
        for exp in EXPERIMENTS:
            m = run(lambda: DSResNetSE(dropout=DROPOUT), train_data, val_data, exp, seed, subset_idx=sub)
            for k in ("acc", "macro_f1", "weighted_f1"):
                per_exp[exp][k].append(m[k])
            print(f"    {exp:<11} acc={m['acc']:.4f} macroF1={m['macro_f1']:.4f}")
    rows = []
    for exp in EXPERIMENTS:
        rows.append({"exp": exp,
                     **{f"{k}_mean": float(np.mean(per_exp[exp][k])) for k in ("acc", "macro_f1", "weighted_f1")},
                     **{f"{k}_std": float(np.std(per_exp[exp][k])) for k in ("acc", "macro_f1", "weighted_f1")}})
    return rows


def setting_small_full(train_data, val_data):
    print("\n##### Setting 2: SMALL student + 100% data (seed 42) #####")
    rows = []
    for exp in EXPERIMENTS:
        m = run(lambda: DSResNetSE(**SMALL_KW), train_data, val_data, exp, 42)
        rows.append({"exp": exp, "acc": m["acc"], "macro_f1": m["macro_f1"], "weighted_f1": m["weighted_f1"]})
        print(f"  {exp:<11} acc={m['acc']:.4f} macroF1={m['macro_f1']:.4f} weightedF1={m['weighted_f1']:.4f}")
    return rows


# ── Table 1 (existing full-data strong) ───────────────────────────────────────────
def load_table1():
    p = DATA / "student" / "results.csv"
    out = {}
    with open(p, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[r["exp"]] = {"acc": float(r["acc"]), "macro_f1": float(r["macro_f1"]),
                             "weighted_f1": float(r["weighted_f1"])}
    return out


# ── Report ─────────────────────────────────────────────────────────────────────────
def write_outputs(t1, t2, t3, params):
    md = []
    md.append(f"### Table 1: Strong student + 100% data  (params {params['strong']:,})")
    md += ["| Method | Val Acc | Macro F1 | Weighted F1 |", "| --- | --- | --- | --- |"]
    for e in EXPERIMENTS:
        r = t1[e]; md.append(f"| {e} | {r['acc']:.4f} | {r['macro_f1']:.4f} | {r['weighted_f1']:.4f} |")

    md.append("\n### Table 2: Strong student + 20% data  (mean +/- std over 3 seeds)")
    md += ["| Method | Val Acc | Macro F1 | Weighted F1 |", "| --- | --- | --- | --- |"]
    for r in t2:
        md.append(f"| {r['exp']} | {r['acc_mean']:.4f}±{r['acc_std']:.4f} | "
                  f"{r['macro_f1_mean']:.4f}±{r['macro_f1_std']:.4f} | "
                  f"{r['weighted_f1_mean']:.4f}±{r['weighted_f1_std']:.4f} |")

    md.append(f"\n### Table 3: Small student + 100% data  (params {params['small']:,}, seed 42)")
    md += ["| Method | Val Acc | Macro F1 | Weighted F1 |", "| --- | --- | --- | --- |"]
    for r in t3:
        md.append(f"| {r['exp']} | {r['acc']:.4f} | {r['macro_f1']:.4f} | {r['weighted_f1']:.4f} |")

    # KD gain over CE-only (macro-F1) per setting — the key question.
    md.append("\n### Macro-F1 gain over CE-only (the key question)")
    md += ["| Method | Strong+100% | Strong+20% | Small+100% |", "| --- | --- | --- | --- |"]
    ce1 = t1["ce_only"]["macro_f1"]
    ce2 = next(r for r in t2 if r["exp"] == "ce_only")["macro_f1_mean"]
    ce3 = next(r for r in t3 if r["exp"] == "ce_only")["macro_f1"]
    for e in EXPERIMENTS:
        g1 = t1[e]["macro_f1"] - ce1
        g2 = next(r for r in t2 if r["exp"] == e)["macro_f1_mean"] - ce2
        g3 = next(r for r in t3 if r["exp"] == e)["macro_f1"] - ce3
        md.append(f"| {e} | {g1:+.4f} | {g2:+.4f} | {g3:+.4f} |")

    (OUT / "limited_ablation_results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n" + "\n".join(md))

    # CSV (long form)
    with open(DATA / "student" / "limited_ablation_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["setting", "method", "acc", "macro_f1", "weighted_f1", "acc_std", "macro_f1_std", "weighted_f1_std"])
        for e in EXPERIMENTS:
            r = t1[e]; w.writerow(["strong_100", e, r["acc"], r["macro_f1"], r["weighted_f1"], "", "", ""])
        for r in t2:
            w.writerow(["strong_20", r["exp"], r["acc_mean"], r["macro_f1_mean"], r["weighted_f1_mean"],
                        r["acc_std"], r["macro_f1_std"], r["weighted_f1_std"]])
        for r in t3:
            w.writerow(["small_100", r["exp"], r["acc"], r["macro_f1"], r["weighted_f1"], "", "", ""])

    make_plot(t1, t2, t3)


def make_plot(t1, t2, t3):
    colors = {"ce_only": "#7f7f7f", "logit_kd": "#1f77b4", "feature_kd": "#ff7f0e", "full_kd": "#2ca02c"}
    fig, axes = plt.subplots(1, 3, figsize=(17, 6), sharey=False)

    panels = [
        ("Strong + 100% data", [t1[e]["macro_f1"] for e in EXPERIMENTS], None),
        ("Strong + 20% data (3-seed)", [next(r for r in t2 if r["exp"] == e)["macro_f1_mean"] for e in EXPERIMENTS],
         [next(r for r in t2 if r["exp"] == e)["macro_f1_std"] for e in EXPERIMENTS]),
        ("Small + 100% data", [next(r for r in t3 if r["exp"] == e)["macro_f1"] for e in EXPERIMENTS], None),
    ]
    for ax, (title, vals, errs) in zip(axes, panels):
        x = np.arange(len(EXPERIMENTS))
        bars = ax.bar(x, vals, color=[colors[e] for e in EXPERIMENTS],
                      yerr=errs, capsize=4 if errs else 0)
        ce = vals[0]
        ax.axhline(ce, color="gray", linestyle=":", linewidth=1.2, label=f"CE-only ({ce:.3f})")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        lo = min(vals) - (max(errs) if errs else 0) - 0.02
        hi = max(vals) + (max(errs) if errs else 0) + 0.02
        ax.set_ylim(lo, hi)
        ax.set_xticks(x); ax.set_xticklabels(EXPERIMENTS, fontsize=8, rotation=15)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, loc="lower right"); ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Macro F1 (FSC val, 31 classes)")
    fig.suptitle("KD Ablation across regimes — does KD help more when data/capacity is limited?\n"
                 "(y-axis zoomed per panel; dotted = CE-only baseline)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "limited_ablation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot -> {OUT / 'limited_ablation.png'}")


def main():
    print(f"Device: {DEVICE}")
    params = {"strong": model_summary(DSResNetSE())["params"],
              "small": model_summary(DSResNetSE(**SMALL_KW))["params"]}
    print(f"strong params {params['strong']:,}  | small params {params['small']:,}")

    train_data, val_data = load_data()

    t1 = load_table1()
    t2 = setting_strong_20pct(train_data, val_data)
    t3 = setting_small_full(train_data, val_data)
    write_outputs(t1, t2, t3, params)
    print("Done.")


if __name__ == "__main__":
    main()
