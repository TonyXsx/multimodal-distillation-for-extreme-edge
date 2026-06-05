"""
Student KD ablation: same DSResNet-SE, four training losses (strong student,
100% data, single seed, ORIGINAL untuned KD defaults T=2 / lam_logit=0.5).

    CE-only   :  L_ce
    Logit-KD  :  L_ce + lam_logit * L_logit
    Feature-KD:  L_ce + lam_feature * L_feature
    Full-KD   :  L_ce + lam_logit * L_logit + lam_feature * L_feature

Shared infra (data, teacher signals, losses, SpecAugment, evaluation, constants)
lives in kd_common.py. The definitive multi-seed, tuned-HP study is
train_student_2x2.py; KD-HP tuning is tune_kd_hparams.py.

Outputs:
  data/student/checkpoints/<exp>_best.pt          (model artifacts)
  outputs/student/main_ablation/results.csv       (source of truth)
  outputs/student/main_ablation/results.md
  outputs/student/main_ablation/student_kd_ablation.png
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
from student_model import DSResNetSE, model_summary   # noqa: E402
from kd_common import (                               # noqa: E402
    DATA, load_data, evaluate, spec_augment, kd_logit_loss, kd_feature_loss,
    EPOCHS, LR, WEIGHT_DECAY, BATCH_SIZE, DROPOUT, LABEL_SMOOTH, SEED, DEVICE,
)

# ── Output paths ──────────────────────────────────────────────────────────────────
CKPT_DIR = DATA / "student" / "checkpoints"
OUT      = PROJECT / "outputs" / "student" / "main_ablation"
for d in (CKPT_DIR, OUT):
    d.mkdir(parents=True, exist_ok=True)

# ── Experiment-specific KD hyperparameters (original untuned defaults) ─────────────
T           = 2.0
LAM_LOGIT   = 0.5
LAM_FEATURE = 1.0

EXPERIMENTS = ["ce_only", "logit_kd", "feature_kd", "full_kd"]
USE_LOGIT   = {"ce_only": False, "logit_kd": True,  "feature_kd": False, "full_kd": True}
USE_FEATURE = {"ce_only": False, "logit_kd": False, "feature_kd": True,  "full_kd": True}


# ── Train one experiment ──────────────────────────────────────────────────────────
def train_experiment(exp, train_data, val_data):
    Xtr, ytr, ztr, ltr = train_data
    Xva, yva, _, _ = val_data
    use_logit, use_feature = USE_LOGIT[exp], USE_FEATURE[exp]

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = DSResNetSE(dropout=DROPOUT).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    ce = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

    n = Xtr.shape[0]
    best_f1, best_state, best_metrics, best_epoch = -1.0, None, None, -1
    history = []

    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        running = 0.0
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb = spec_augment(Xtr[idx].to(DEVICE))
            yb = ytr[idx].to(DEVICE)
            opt.zero_grad()
            z_s, logits_s = model(xb)
            loss = ce(logits_s, yb)
            if use_logit:
                loss = loss + LAM_LOGIT * kd_logit_loss(logits_s, ltr[idx].to(DEVICE), T)
            if use_feature:
                loss = loss + LAM_FEATURE * kd_feature_loss(z_s, ztr[idx].to(DEVICE).float())
            loss.backward()
            opt.step()
            running += loss.item() * len(idx)
        sched.step()

        m = evaluate(model, Xva, yva)
        history.append({"epoch": epoch, "train_loss": running / n, **m})
        if m["macro_f1"] > best_f1:
            best_f1 = m["macro_f1"]
            best_metrics = m
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:
            print(f"    epoch {epoch+1:3d}/{EPOCHS}  loss={running/n:.4f}  "
                  f"val_acc={m['acc']:.4f}  macroF1={m['macro_f1']:.4f}")

    # Save best-by-val-macroF1 checkpoint.
    torch.save({"exp": exp, "state_dict": best_state, "best_epoch": best_epoch,
                "metrics": best_metrics, "history": history},
               CKPT_DIR / f"{exp}_best.pt")
    return best_metrics, best_epoch, history


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")
    sz = model_summary(DSResNetSE())
    print(f"Student params: {sz['params']:,}  | FP32 {sz['fp32_mb']:.2f} MB  | INT8(est) {sz['int8_mb']:.2f} MB\n")

    train_data, val_data = load_data()

    results = []
    for exp in EXPERIMENTS:
        print(f"\n=== {exp} ===")
        metrics, best_epoch, _ = train_experiment(exp, train_data, val_data)
        print(f"  BEST @epoch {best_epoch+1}: acc={metrics['acc']:.4f}  "
              f"macroF1={metrics['macro_f1']:.4f}  weightedF1={metrics['weighted_f1']:.4f}")
        results.append({"exp": exp, **metrics,
                        "params": sz["params"], "fp32_mb": round(sz["fp32_mb"], 2),
                        "int8_mb": round(sz["int8_mb"], 2)})

    # ── results CSV (source of truth) + MD ────────────────────────────────────────
    csv_path = OUT / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["exp", "acc", "macro_f1", "weighted_f1",
                                          "params", "fp32_mb", "int8_mb"])
        w.writeheader()
        w.writerows(results)
    print(f"\nResults CSV -> {csv_path}")

    md = ["| Experiment | Val Acc | Macro F1 | Weighted F1 | Params | FP32 MB | INT8 MB |",
          "| ---------- | ------- | -------- | ----------- | ------ | ------- | ------- |"]
    for r in results:
        md.append(f"| {r['exp']} | {r['acc']:.4f} | {r['macro_f1']:.4f} | {r['weighted_f1']:.4f} "
                  f"| {r['params']:,} | {r['fp32_mb']} | {r['int8_mb']} |")
    (OUT / "results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n" + "\n".join(md))

    make_plot(results)


def make_plot(results):
    exps = [r["exp"] for r in results]
    metrics = [("acc", "Val Accuracy", "#1f77b4"),
               ("macro_f1", "Macro F1", "#ff7f0e"),
               ("weighted_f1", "Weighted F1", "#2ca02c")]

    vals_all = [r[m] for r in results for m, _, _ in metrics]
    lo, hi = min(vals_all) - 0.01, max(vals_all) + 0.008

    fig, ax = plt.subplots(figsize=(11, 6.5))
    x = np.arange(len(exps))
    w = 0.26
    for j, (key, label, color) in enumerate(metrics):
        vals = [r[key] for r in results]
        bars = ax.bar(x + (j - 1) * w, vals, w, label=label, color=color)
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() - 0.0009,
                    f"{b.get_height():.4f}", ha="center", va="top",
                    fontsize=8, rotation=90, color="white", fontweight="bold")

    ce_acc = results[0]["acc"]
    ax.axhline(ce_acc, color="gray", linestyle=":", linewidth=1.3,
               label=f"CE-only acc ({ce_acc:.4f})")

    ax.set_xticks(x)
    ax.set_xticklabels(exps, fontsize=10)
    ax.set_ylim(lo, hi)
    ax.set_ylabel("Score (FSC validation, 31 classes)")
    ax.set_title("Student KD Ablation — DSResNet-SE (379K params, FP32 1.45 MB) + SpecAugment + label-smooth\n"
                 "audio-only student  ·  teacher = Qwen2.5-Omni-3B B2 bottleneck(64) + logits  "
                 "·  (y-axis zoomed)", fontsize=10)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="y", alpha=0.35)

    plot_path = OUT / "student_kd_ablation.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot        -> {plot_path}")
    print("Done.")


if __name__ == "__main__":
    main()
